# stt_cli_client.py

from difflib import SequenceMatcher
from collections import deque
import argparse
import string
import shutil
import time
import sys
import os

from RealtimeSTT import AudioToTextRecorderClient
from RealtimeSTT import AudioInput

from colorama import init, Fore, Style
init()

DEFAULT_CONTROL_URL = "ws://127.0.0.1:8011"
DEFAULT_DATA_URL = "ws://127.0.0.1:8012"

recording_indicator = "🔴"

console_width = shutil.get_terminal_size().columns

post_speech_silence_duration = 1.0  # Will be overridden by CLI arg
unknown_sentence_detection_pause = 1.3
mid_sentence_detection_pause = 3.0
end_of_sentence_detection_pause = 0.7
hard_break_even_on_background_noise = 3.0
hard_break_even_on_background_noise_min_texts = 3
hard_break_even_on_background_noise_min_similarity = 0.99
hard_break_even_on_background_noise_min_chars = 15
prev_text = ""
text_time_deque = deque()

def main():
    global prev_text, post_speech_silence_duration, unknown_sentence_detection_pause
    global mid_sentence_detection_pause, end_of_sentence_detection_pause
    global hard_break_even_on_background_noise, hard_break_even_on_background_noise_min_texts
    global hard_break_even_on_background_noise_min_similarity, hard_break_even_on_background_noise_min_chars

    parser = argparse.ArgumentParser(description="STT Client")

    # Add input device argument
    parser.add_argument("-i", "--input-device", type=int, metavar="INDEX",
                        help="Audio input device index (use -l to list devices)")
    parser.add_argument("-l", "--language", default="en", metavar="LANG",
                        help="Language to be used (default: en)")
    parser.add_argument("-sed", "--speech-end-detection", action="store_true",
                        help="Usage of intelligent speech end detection")
    parser.add_argument("-D", "--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("-n", "--norealtime", action="store_true",
                        help="Disable real-time output")
    parser.add_argument("-W", "--write", metavar="FILE",
                        help="Save recorded audio to a WAV file")
    parser.add_argument("-s", "--set", nargs=2, metavar=('PARAM', 'VALUE'), action='append',
                        help="Set a recorder parameter (can be used multiple times)")
    parser.add_argument("-m", "--method", nargs='+', metavar='METHOD', action='append',
                        help="Call a recorder method with optional arguments")
    parser.add_argument("-g", "--get", nargs=1, metavar='PARAM', action='append',
                        help="Get a recorder parameter's value (can be used multiple times)")
    parser.add_argument("-c", "--continous", action="store_true",
                        help="Continuously transcribe speech without exiting")
    parser.add_argument("-L", "--list", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--control", "--control_url", default=DEFAULT_CONTROL_URL,
                        help="STT Control WebSocket URL")
    parser.add_argument("--data", "--data_url", default=DEFAULT_DATA_URL,
                        help="STT Data WebSocket URL")
    parser.add_argument("--post-silence", type=float, default=1.0,
                      help="Post speech silence duration in seconds (default: 1.0)")
    parser.add_argument("--unknown-pause", type=float, default=1.3,
                      help="Unknown sentence detection pause in seconds (default: 1.3)")
    parser.add_argument("--mid-pause", type=float, default=3.0,
                      help="Mid sentence detection pause in seconds (default: 3.0)")
    parser.add_argument("--end-pause", type=float, default=0.7,
                      help="End of sentence detection pause in seconds (default: 0.7)")
    parser.add_argument("--hard-break", type=float, default=3.0,
                      help="Hard break threshold in seconds (default: 3.0)")
    parser.add_argument("--min-texts", type=int, default=3,
                      help="Minimum texts for hard break (default: 3)")
    parser.add_argument("--min-similarity", type=float, default=0.99,
                      help="Minimum text similarity for hard break (default: 0.99)")
    parser.add_argument("--min-chars", type=int, default=15,
                      help="Minimum characters for hard break (default: 15)")

    args = parser.parse_args()

    # Add this block after parsing args:
    if args.list:
        audio_input = AudioInput()
        audio_input.list_devices()
        return

    # Update globals with CLI values
    post_speech_silence_duration = args.post_silence
    unknown_sentence_detection_pause = args.unknown_pause
    mid_sentence_detection_pause = args.mid_pause
    end_of_sentence_detection_pause = args.end_pause
    hard_break_even_on_background_noise = args.hard_break
    hard_break_even_on_background_noise_min_texts = args.min_texts
    hard_break_even_on_background_noise_min_similarity = args.min_similarity
    hard_break_even_on_background_noise_min_chars = args.min_chars

    # Check if output is being redirected
    if not os.isatty(sys.stdout.fileno()):
        file_output = sys.stdout
    else:
        file_output = None

    def clear_line():
        if file_output:
            sys.stderr.write('\r\033[K')
        else:
            print('\r\033[K', end="", flush=True)

    def write(text):
        if file_output:
            sys.stderr.write(text)
            sys.stderr.flush()
        else:
            print(text, end="", flush=True)

    def on_realtime_transcription_update(text):
        global post_speech_silence_duration, prev_text, text_time_deque
    
        def set_post_speech_silence_duration(duration: float):
            global post_speech_silence_duration
            post_speech_silence_duration = duration
            client.set_parameter("post_speech_silence_duration", duration)

        def preprocess_text(text):
            text = text.lstrip()
            if text.startswith("..."):
                text = text[3:]
            text = text.lstrip()
            if text:
                text = text[0].upper() + text[1:]
            return text

        def ends_with_ellipsis(text: str):
            if text.endswith("..."):
                return True
            if len(text) > 1 and text[:-1].endswith("..."):
                return True
            return False

        def sentence_end(text: str):
            sentence_end_marks = ['.', '!', '?', '。']
            if text and text[-1] in sentence_end_marks:
                return True
            return False

        if not args.norealtime:
            text = preprocess_text(text)

            if args.speech_end_detection:
                if ends_with_ellipsis(text):
                    if not post_speech_silence_duration == mid_sentence_detection_pause:
                        set_post_speech_silence_duration(mid_sentence_detection_pause)
                        if args.debug: print(f"RT: post_speech_silence_duration for {text} (...): {post_speech_silence_duration}")
                elif sentence_end(text) and sentence_end(prev_text) and not ends_with_ellipsis(prev_text):
                    if not post_speech_silence_duration == end_of_sentence_detection_pause:
                        set_post_speech_silence_duration(end_of_sentence_detection_pause)
                        if args.debug: print(f"RT: post_speech_silence_duration for {text} (.!?): {post_speech_silence_duration}")
                else:
                    if not post_speech_silence_duration == unknown_sentence_detection_pause:
                        set_post_speech_silence_duration(unknown_sentence_detection_pause)
                        if args.debug: print(f"RT: post_speech_silence_duration for {text} (???): {post_speech_silence_duration}")
                
                prev_text = text

                # transtext = text.translate(str.maketrans('', '', string.punctuation))
                
                # Append the new text with its timestamp
                current_time = time.time()
                text_time_deque.append((current_time, text))

                # Remove texts older than hard_break_even_on_background_noise seconds
                while text_time_deque and text_time_deque[0][0] < current_time - hard_break_even_on_background_noise:
                    text_time_deque.popleft()

                # Check if at least hard_break_even_on_background_noise_min_texts texts have arrived within the last hard_break_even_on_background_noise seconds
                if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
                    texts = [t[1] for t in text_time_deque]
                    first_text = texts[0]
                    last_text = texts[-1]

                    # Compute the similarity ratio between the first and last texts
                    similarity = SequenceMatcher(None, first_text, last_text).ratio()

                    if similarity > hard_break_even_on_background_noise_min_similarity and len(first_text) > hard_break_even_on_background_noise_min_chars:
                        client.call_method("stop")

            clear_line()

            words = text.split()
            last_chars = ""
            available_width = console_width - 5
            for word in reversed(words):
                if len(last_chars) + len(word) + 1 > available_width:
                    break
                last_chars = word + " " + last_chars
            last_chars = last_chars.strip()

            colored_text = f"{Fore.YELLOW}{last_chars}{Style.RESET_ALL}{recording_indicator}\b\b"
            write(colored_text)

    client = AudioToTextRecorderClient(
        language=args.language,
        control_url=args.control,
        data_url=args.data,
        debug_mode=args.debug,
        on_realtime_transcription_update=on_realtime_transcription_update,
        use_microphone=True,
        input_device_index=args.input_device,  # Pass input device index
        output_wav_file = args.write or None,
    )

    # Process command-line parameters
    if args.set:
        for param, value in args.set:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string if not a number
            client.set_parameter(param, value)

    if args.get:
        for param_list in args.get:
            param = param_list[0]
            value = client.get_parameter(param)
            if value is not None:
                print(f"Parameter {param} = {value}")

    if args.method:
        for method_call in args.method:
            method = method_call[0]
            args_list = method_call[1:] if len(method_call) > 1 else []
            client.call_method(method, args=args_list)

    # Start transcription
    try:
        while True:
            if not client._recording:
                print("Recording stopped due to an error.", file=sys.stderr)
                break
            
            if not file_output:
                print(recording_indicator, end="", flush=True)
            else:
                sys.stderr.write(recording_indicator)
                sys.stderr.flush()
                
            text = client.text()

            if text and client._recording and client.is_running:
                if file_output:
                    print(text, file=file_output)
                    sys.stderr.write('\r\033[K')
                    sys.stderr.write(f'{text}')
                else:
                    print('\r\033[K', end="", flush=True)
                    print(f'{text}', end="", flush=True)
                if not args.continous:
                    break
            else:
                time.sleep(0.1)
            
            if args.continous:
                print()
            prev_text = ""
    except KeyboardInterrupt:
        print('\r\033[K', end="", flush=True)
    finally:
        client.shutdown()

if __name__ == "__main__":
    main()
