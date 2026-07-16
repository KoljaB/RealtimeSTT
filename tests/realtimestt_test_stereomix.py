EXTENDED_LOGGING = False

def main():

    from install_packages import check_and_install_packages
    check_and_install_packages([
        {
            'import_name': 'rich',
        }
    ])

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    import os
    import sys
    import threading
    import time
    import pyaudio
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from colorama import Fore, Style, init as colorama_init

    from RealtimeSTT import AudioToTextRecorder 

    # Configuration Constants
    LOOPBACK_DEVICE_NAME = "stereomix"
    LOOPBACK_DEVICE_HOST_API = 0
    BUFFER_SIZE = 512 
    AUDIO_FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    console = Console()
    console.print("System initializing, please wait")

    colorama_init()

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.2
    unknown_sentence_detection_pause = 0.5
    mid_sentence_detection_pause = 1

    prev_text = ""

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        # Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    def text_detected(text):
        nonlocal prev_text, displayed_text, rich_text_stored

        text = preprocess_text(text)

        sentence_end_marks = ['.', '!', '?', '。']
        midsentence_marks = ['…', '-', '(']
        if text.endswith("...") or text and text[-1] in midsentence_marks:
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        # Build Rich Text with alternating colors
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")

        # If the current text is not a sentence-ending, display it in real-time
        if text:
            rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        nonlocal recorder, full_sentences, prev_text
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]  # Remove ellipsis

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'use_microphone': False,
        'model': 'large-v2',
        'input_device_index': None,  # To be set after finding the device
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 2.0,        
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.01,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': False,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 1,
        'no_log_file': True,
        'initial_prompt': "Use ellipses for incomplete sentences like: I went to the..."
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    def find_stereo_mix_index():
        nonlocal audio
        devices_info = ""
        for i in range(audio.get_device_count()):
            dev = audio.get_device_info_by_index(i)
            devices_info += f"{dev['index']}: {dev['name']} (hostApi: {dev['hostApi']})\n"

            if (LOOPBACK_DEVICE_NAME.lower() in dev['name'].lower()
                    and dev['hostApi'] == LOOPBACK_DEVICE_HOST_API):
                return dev['index'], devices_info

        return None, devices_info

    device_index, devices_info = find_stereo_mix_index()
    if device_index is None:
        live.stop()
        console.print("[bold red]Stereo Mix device not found. Available audio devices are:\n[/bold red]")
        console.print(devices_info, style="red")
        audio.terminate()
        sys.exit(1)
    else:
        recorder_config['input_device_index'] = device_index
        console.print(f"Using audio device index {device_index} for Stereo Mix.", style="green")

    # Initialize the recorder
    recorder = AudioToTextRecorder(**recorder_config)

    # Initialize Live Display with waiting message
    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    # Define the recording thread
    def recording_thread():
        nonlocal recorder
        stream = audio.open(format=AUDIO_FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=BUFFER_SIZE,
                            input_device_index=recorder_config['input_device_index'])

        try:
            while not stop_event.is_set():
                data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
                recorder.feed_audio(data)
        except Exception as e:
            console.print(f"[bold red]Error in recording thread: {e}[/bold red]")
        finally:
            console.print(f"[bold red]Stopping stream[/bold red]")
            stream.stop_stream()
            stream.close()

    # Define the stop event
    stop_event = threading.Event()

    # Start the recording thread
    thread = threading.Thread(target=recording_thread, daemon=True)
    thread.start()

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        console.print("[bold red]\nTranscription stopped by user. Exiting...[/bold red]")
    finally:
        print("live stop")
        live.stop()

        print("setting stop event")
        stop_event.set()

        print("thread join")
        thread.join()

        print("recorder stop")
        recorder.stop()

        print("audio terminate")
        audio.terminate()

        print("sys exit ")
        sys.exit(0)

if __name__ == '__main__':
    main()
