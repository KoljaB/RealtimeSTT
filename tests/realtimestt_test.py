EXTENDED_LOGGING = False

# set to 0 to deactivate writing to the focused window
WRITE_TO_KEYBOARD_INTERVAL = 0.002
CLIPBOARD_RESTORE_DELAY = 0.1

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Start the realtime Speech-to-Text (STT) test with various configuration options.')

    parser.add_argument('-m', '--model', type=str, # no default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, # no default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument(
        '-u', '--ultrafast-rt-model', '--ultrafast_realtime_model_type',
        dest='ultrafast_rt_model', type=str,
        help='Optional lower-latency realtime model. Safely aligned words from this lane are appended to the accurate realtime text.'
    )
    parser.add_argument(
        '--ultrafast-rt-engine',
        dest='ultrafast_rt_engine', type=str,
        help='Optional transcription engine for the ultrafast realtime lane.'
    )
    parser.add_argument(
        '--ultrafast-rt-engine-options',
        dest='ultrafast_rt_engine_options', type=str,
        help='Optional JSON object with engine-specific options for the ultrafast lane.'
    )
    parser.add_argument(
        '--ultrafast-max-tail-words',
        dest='ultrafast_max_tail_words', type=int, default=5,
        help='Maximum safely aligned ultrafast words appended to the accurate text. Default is 5.'
    )
    parser.add_argument('-l', '--lang', '--language', dest='language', type=str, # no default='en',
                help='Language code forwarded to AudioToTextRecorder. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')
    
    parser.add_argument('-d', '--root', type=str, # no default=None,
                help='Root directory where the Whisper models are downloaded to.')

    from install_packages import check_and_install_packages
    check_and_install_packages([
        {
            'import_name': 'rich',
        },
        {
            'import_name': 'pyautogui',
        },
        {
            'import_name': 'pyperclip',
        }
    ])

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    console.print("System initializing, please wait")

    import os
    import sys
    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama
    import pyautogui
    import pyperclip
    import json
    import time

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.9
    unknown_sentence_detection_pause = 1.2
    mid_sentence_detection_pause = 2.5

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    prev_text = ""

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        #  Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text


    def text_detected(update):
        global prev_text, displayed_text, rich_text_stored

        accurate_text = ''
        ultrafast_suffix = ''
        if hasattr(update, 'slow_text'):
            accurate_text = preprocess_text(update.slow_text or '')
            raw_suffix = ' '.join((update.ultrafast_suffix or '').split())
            ultrafast_suffix = (' ' if accurate_text and raw_suffix else '') + raw_suffix
            text = preprocess_text(update.text or update.slow_text or '')
        else:
            stable_text = ''
            if hasattr(update, 'display_text'):
                stable_text = preprocess_text(update.stable_text or '')
                update = update.display_text or update.raw_observation_text
            text = preprocess_text(update)
            if text.casefold().startswith(stable_text.casefold()):
                accurate_text = text[:len(stable_text)]
                ultrafast_suffix = text[len(stable_text):]
            else:
                ultrafast_suffix = text

        sentence_end_marks = ['.', '!', '?', '。']
        if text.endswith('...'):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        rich_text = Text()
        for index, sentence in enumerate(full_sentences):
            style = 'yellow' if index % 2 == 0 else 'blue'
            rich_text += Text(sentence, style=style) + Text(' ')

        if text:
            rich_text += Text(accurate_text, style='white')
            rich_text += Text(ultrafast_suffix, style='grey50')

        new_displayed_text = rich_text.plain
        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title='[bold green]Live Transcription[/bold green]', border_style='bold green')
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

        if WRITE_TO_KEYBOARD_INTERVAL:
            write_text_to_keyboard(f"{text} ")

    def write_text_to_keyboard(text):
        previous_clipboard = pyperclip.paste()
        try:
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(CLIPBOARD_RESTORE_DELAY)
        finally:
            pyperclip.copy(previous_clipboard)

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'large-v2', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'download_root': None, # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        # 'input_device_index': 1,
        'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_punctuation_split_marks': 'sentence',
        'realtime_processing_pause': 0.1,
        'on_realtime_text_stabilization_update': text_detected,
        'ultrafast_realtime_max_tail_words': 5,
        #'on_realtime_transcription_stabilized': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'realtime_transcription_use_syllable_boundaries': True,
        'realtime_boundary_detector_sensitivity': 0.6,
        'realtime_boundary_followup_delays': (0.05, 0.2),
        'beam_size': 5,
        'beam_size_realtime': 3,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,        
        'no_log_file': True,
        'initial_prompt_realtime': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
    }

    args = parser.parse_args()
    if args.model is not None:
        recorder_config['model'] = args.model
        print(f"Argument 'model' set to {recorder_config['model']}")
    if args.rt_model is not None:
        recorder_config['realtime_model_type'] = args.rt_model
        print(f"Argument 'realtime_model_type' set to {recorder_config['realtime_model_type']}")
    if args.ultrafast_rt_model is not None:
        if args.ultrafast_max_tail_words < 1:
            parser.error('--ultrafast-max-tail-words must be at least 1')
        recorder_config['ultrafast_realtime_model_type'] = args.ultrafast_rt_model
        recorder_config['ultrafast_realtime_max_tail_words'] = args.ultrafast_max_tail_words
        recorder_config['on_realtime_text_stabilization_update'] = None
        recorder_config['on_realtime_transcription_merge_update'] = text_detected
        if args.ultrafast_rt_engine is not None:
            recorder_config['ultrafast_realtime_transcription_engine'] = args.ultrafast_rt_engine
        if args.ultrafast_rt_engine_options is not None:
            try:
                engine_options = json.loads(args.ultrafast_rt_engine_options)
            except json.JSONDecodeError as error:
                parser.error(f'--ultrafast-rt-engine-options must be valid JSON: {error}')
            if not isinstance(engine_options, dict):
                parser.error('--ultrafast-rt-engine-options must decode to a JSON object')
            recorder_config['ultrafast_realtime_transcription_engine_options'] = engine_options
        print(f"Argument 'ultrafast_realtime_model_type' set to {recorder_config['ultrafast_realtime_model_type']}")
        print(f"Argument 'ultrafast_realtime_max_tail_words' set to {recorder_config['ultrafast_realtime_max_tail_words']}")
    if args.language is not None:
        recorder_config['language'] = args.language
        print(f"Argument 'language' set to {recorder_config['language']}")
    if args.root is not None:
        recorder_config['download_root'] = args.root
        print(f"Argument 'download_root' set to {recorder_config['download_root']}")

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)
