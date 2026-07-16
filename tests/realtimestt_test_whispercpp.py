EXTENDED_LOGGING = False

# Set to 0 to deactivate writing to keyboard.
# Try lower values like 0.002 first, use higher values like 0.05 in case it fails.
WRITE_TO_KEYBOARD_INTERVAL = 0.002
FINAL_MODEL_PROFILES = {
    "fast": {
        "model": "base.en-q5_1",
        "beam_size": 3,
        "description": "fast final transcription, better than tiny, low CPU cost",
    },
    "balanced": {
        "model": "small.en-q5_1",
        "beam_size": 3,
        "description": "good CPU default, noticeably better than tiny without medium-size cost",
    },
    "accurate": {
        "model": "small.en",
        "beam_size": 5,
        "description": "slower final transcription with better quality",
    },
}
DEFAULT_PROFILE = "balanced"
DEFAULT_FINAL_MODEL = FINAL_MODEL_PROFILES[DEFAULT_PROFILE]["model"]
DEFAULT_REALTIME_MODEL = "tiny.en"


if __name__ == '__main__':

    import argparse
    import os
    import sys

    default_threads = max(1, min(8, os.cpu_count() or 4))
    default_model_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "test-model-cache",
            "pywhispercpp",
        )
    )

    parser = argparse.ArgumentParser(
        description=(
            "Start the realtime Speech-to-Text (STT) test using the "
            "whisper.cpp transcription engine."
        )
    )

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help=(
            "whisper.cpp model name or ggml model path for final transcription. "
            f"Default is {DEFAULT_FINAL_MODEL} from --profile {DEFAULT_PROFILE}."
        ),
    )

    parser.add_argument(
        '--profile',
        choices=sorted(FINAL_MODEL_PROFILES),
        default=DEFAULT_PROFILE,
        help=(
            "Final transcription quality/speed profile. "
            "Use --model or --beam-size to override profile values. "
            f"Default is {DEFAULT_PROFILE}."
        ),
    )

    parser.add_argument(
        '-r',
        '--rt-model',
        '--realtime_model_type',
        type=str,
        help=(
            "whisper.cpp model name or ggml model path for real-time transcription. "
            f"Default is {DEFAULT_REALTIME_MODEL}."
        ),
    )

    parser.add_argument(
        '-l',
        '--lang',
        '--language',
        type=str,
        help=(
            "Language code for transcription. Leave empty for auto-detection. "
            "Default is en."
        ),
    )

    parser.add_argument(
        '-d',
        '--root',
        type=str,
        help=(
            "Root directory where pywhispercpp downloads or looks up ggml models. "
            f"Default is {default_model_root}."
        ),
    )

    parser.add_argument(
        '--no-keyboard',
        action='store_true',
        help="Do not type finalized transcriptions into the active window.",
    )

    parser.add_argument(
        '--keyboard-interval',
        type=float,
        default=WRITE_TO_KEYBOARD_INTERVAL,
        help=(
            "Interval used by pyautogui.write for finalized text. "
            "Set to 0 to disable keyboard output."
        ),
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=default_threads,
        help=(
            "Number of whisper.cpp CPU threads for final transcription. "
            f"Default is {default_threads}."
        ),
    )

    parser.add_argument(
        '--rt-threads',
        type=int,
        default=default_threads,
        help=(
            "Number of whisper.cpp CPU threads for real-time transcription. "
            f"Default is {default_threads}."
        ),
    )

    parser.add_argument(
        '--beam-size',
        type=int,
        default=None,
        help=(
            "Beam size for final transcription. "
            "Default comes from --profile."
        ),
    )

    parser.add_argument(
        '--rt-beam-size',
        type=int,
        default=1,
        help=(
            "Beam size for real-time transcription. Default is 1 for faster "
            "greedy decoding on CPU."
        ),
    )

    parser.add_argument(
        '--rt-pause',
        type=float,
        default=1,
        help=(
            "Seconds between real-time transcription attempts. Default is 1. "
            "Lower values can waste CPU if inference is slower than the pause."
        ),
    )

    parser.add_argument(
        '--rt-audio-ctx',
        type=int,
        default=0,
        help=(
            "Optional whisper.cpp audio context override for real-time "
            "transcription. 0 keeps the model default."
        ),
    )

    parser.add_argument(
        '--realtime-context',
        action='store_true',
        help=(
            "Allow whisper.cpp to reuse previous text context during real-time "
            "transcription. Disabled by default for speed."
        ),
    )

    parser.add_argument(
        '--realtime-multi-segment',
        action='store_true',
        help=(
            "Allow multiple whisper.cpp segments during real-time transcription. "
            "Disabled by default for speed."
        ),
    )

    parser.add_argument(
        '--realtime-prompt',
        action='store_true',
        help=(
            "Use the incomplete-sentence prompt during real-time transcription. "
            "Disabled by default because prompts cost CPU time."
        ),
    )

    args = parser.parse_args()
    keyboard_interval = 0 if args.no_keyboard else args.keyboard_interval
    final_profile = FINAL_MODEL_PROFILES[args.profile]
    final_model = args.model or final_profile["model"]
    final_beam_size = (
        args.beam_size if args.beam_size is not None else final_profile["beam_size"]
    )

    from install_packages import check_and_install_packages
    check_and_install_packages([
        {
            'import_name': 'rich',
        },
        {
            'import_name': 'pyautogui',
        },
        {
            'import_name': 'pywhispercpp',
        },
    ])

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel

    console = Console()
    console.print("System initializing with whisper.cpp, please wait")
    console.print(
        f"Final profile: {args.profile} "
        f"({final_model}, beam {final_beam_size}) - {final_profile['description']}"
    )

    from RealtimeSTT import AudioToTextRecorder
    import colorama
    import pyautogui

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    colorama.init()

    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    prev_text = ""

    def preprocess_text(text):
        text = text.lstrip()

        if text.startswith("..."):
            text = text[3:]

        text = text.lstrip()

        if text:
            text = text[0].upper() + text[1:]

        return text

    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored

        text = preprocess_text(text)

        sentence_end_marks = ['.', '!', '?', '\u3002']
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")

        if text:
            rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(
                rich_text,
                title="[bold green]Live Transcription - whisper.cpp[/bold green]",
                border_style="bold green",
            )
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        global recorder, full_sentences, prev_text

        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        if not text:
            return

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        if keyboard_interval:
            pyautogui.write(f"{text} ", interval=keyboard_interval)

    realtime_transcribe_options = {
        'single_segment': not args.realtime_multi_segment,
        'no_context': not args.realtime_context,
        'print_timestamps': False,
    }
    if args.rt_audio_ctx:
        realtime_transcribe_options['audio_ctx'] = args.rt_audio_ctx

    realtime_prompt = None
    if args.realtime_prompt:
        realtime_prompt = (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )

    recorder_config = {
        'spinner': False,
        'model': final_model,
        'transcription_engine': 'whisper_cpp',
        'transcription_engine_options': {
            'model': {
                'n_threads': args.threads,
                'redirect_whispercpp_logs_to': None,
            },
            'transcribe': {
                'print_timestamps': False,
            },
        },
        'download_root': default_model_root,
        'realtime_model_type': DEFAULT_REALTIME_MODEL,
        'realtime_transcription_engine': 'whisper_cpp',
        'realtime_transcription_engine_options': {
            'model': {
                'n_threads': args.rt_threads,
                'redirect_whispercpp_logs_to': None,
            },
            'transcribe': realtime_transcribe_options,
        },
        'language': 'en',
        'device': 'cpu',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': args.rt_pause,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': final_beam_size,
        'beam_size_realtime': args.rt_beam_size,
        'batch_size': 0,
        'realtime_batch_size': 0,
        #'realtime_processing_pause': 1,
        'realtime_transcription_use_syllable_boundaries': True,
        'realtime_boundary_detector_sensitivity': 0.6,
        'realtime_boundary_followup_delays': (0.5),
        'no_log_file': True,
        'initial_prompt': (
            "Technical terms that may appear: RealtimeSTT, realtime transcription, "
            "speech-to-text, whisper.cpp, pywhispercpp, faster-whisper, CPU, GPU."
        ),
        'initial_prompt_realtime': realtime_prompt,
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
        'normalize_audio': True,
    }

    if args.model is not None:
        print(f"Argument 'model' set to {recorder_config['model']}")
    if args.rt_model is not None:
        recorder_config['realtime_model_type'] = args.rt_model
        print(f"Argument 'realtime_model_type' set to {recorder_config['realtime_model_type']}")
    if args.lang is not None:
        recorder_config['language'] = args.lang
        print(f"Argument 'language' set to {recorder_config['language']}")
    if args.root is not None:
        recorder_config['download_root'] = args.root
        print(f"Argument 'download_root' set to {recorder_config['download_root']}")

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)

    initial_text = Panel(
        Text("Say something...", style="cyan bold"),
        title="[bold yellow]Waiting for Input - whisper.cpp[/bold yellow]",
        border_style="bold yellow",
    )
    live.update(initial_text)

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        recorder.shutdown()
        exit(0)
