IS_DEBUG = False

import os
import sys
import threading
import queue
import time
from collections import deque
from difflib import SequenceMatcher
from install_packages import check_and_install_packages

# Check and install required packages
check_and_install_packages([
    {'import_name': 'rich'},
    {'import_name': 'openai'},
    {'import_name': 'colorama'},
    {'import_name': 'RealtimeSTT'},
    # Add any other required packages here
])

EXTENDED_LOGGING = False

if __name__ == '__main__':

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

    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama
    from openai import OpenAI
    # import ollama

    # Initialize OpenAI client for Ollama    
    client = OpenAI(
        # base_url='http://127.0.0.1:11434/v1/', # ollama
        base_url='http://127.0.0.1:1234/v1/', # lm_studio
        api_key='ollama',  # required but ignored
    )

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    # Initialize a thread-safe queue
    text_queue = queue.Queue()

    # Variables for managing displayed text
    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""
    text_time_deque = deque()

    rapid_sentence_end_detection = 0.4
    end_of_sentence_detection_pause = 1.2
    unknown_sentence_detection_pause = 1.8
    mid_sentence_detection_pause = 2.4
    hard_break_even_on_background_noise = 3.0
    hard_break_even_on_background_noise_min_texts = 3
    hard_break_even_on_background_noise_min_chars = 15
    hard_break_even_on_background_noise_min_similarity = 0.99
    relisten_on_abrupt_stop = True

    abrupt_stop = False

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    prev_text = ""

    speech_finished_cache = {}

    def is_speech_finished(text):
        # Check if the result is already in the cache
        if text in speech_finished_cache:
            if IS_DEBUG:
                print(f"Cache hit for: '{text}'")
            return speech_finished_cache[text]
        
        user_prompt = (
            "Please reply with only 'c' if the following text is a complete thought (a sentence that stands on its own), "
            "or 'i' if it is not finished. Do not include any additional text in your reply. "
            "Consider a full sentence to have a clear subject, verb, and predicate or express a complete idea. "
            "Examples:\n"
            "- 'The sky is blue.' is complete (reply 'c').\n"
            "- 'When the sky' is incomplete (reply 'i').\n"
            "- 'She walked home.' is complete (reply 'c').\n"
            "- 'Because he' is incomplete (reply 'i').\n"
            f"\nText: {text}"
        )

        response = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1,
            temperature=0.0,  # Set temperature to 0 for deterministic output
        )

        if IS_DEBUG:
            print(f"t:'{response.choices[0].message.content.strip().lower()}'", end="", flush=True)

        reply = response.choices[0].message.content.strip().lower()
        result = reply == 'c'

        # Cache the result
        speech_finished_cache[text] = result

        return result

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

    def text_detected(text):
        """
        Enqueue the detected text for processing.
        """
        text_queue.put(text)


    def process_queue():
        global recorder, full_sentences, prev_text, displayed_text, rich_text_stored, text_time_deque, abrupt_stop

        # Initialize a deque to store texts with their timestamps
        while True:
            try:
                text = text_queue.get(timeout=1)  # Wait for text or timeout after 1 second
            except queue.Empty:
                continue  # No text to process, continue looping

            if text is None:
                # Sentinel value to indicate thread should exit
                break

            text = preprocess_text(text)
            current_time = time.time()

            sentence_end_marks = ['.', '!', '?', '。'] 
            if text.endswith("..."):
                if not recorder.post_speech_silence_duration == mid_sentence_detection_pause:
                    recorder.post_speech_silence_duration = mid_sentence_detection_pause
                    if IS_DEBUG: print(f"RT: post_speech_silence_duration: {recorder.post_speech_silence_duration}")
            elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
                if not recorder.post_speech_silence_duration == end_of_sentence_detection_pause:
                    recorder.post_speech_silence_duration = end_of_sentence_detection_pause
                    if IS_DEBUG: print(f"RT: post_speech_silence_duration: {recorder.post_speech_silence_duration}")
            else:
                if not recorder.post_speech_silence_duration == unknown_sentence_detection_pause:
                    recorder.post_speech_silence_duration = unknown_sentence_detection_pause
                    if IS_DEBUG: print(f"RT: post_speech_silence_duration: {recorder.post_speech_silence_duration}")

            prev_text = text
            
            import string
            transtext = text.translate(str.maketrans('', '', string.punctuation))
            
            if is_speech_finished(transtext):
                if not recorder.post_speech_silence_duration == rapid_sentence_end_detection:
                    recorder.post_speech_silence_duration = rapid_sentence_end_detection
                    if IS_DEBUG: print(f"RT: {transtext} post_speech_silence_duration: {recorder.post_speech_silence_duration}")

            # Append the new text with its timestamp
            text_time_deque.append((current_time, text))

            # Remove texts older than 1 second
            while text_time_deque and text_time_deque[0][0] < current_time - hard_break_even_on_background_noise:
                text_time_deque.popleft()

            # Check if at least 3 texts have arrived within the last full second
            if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
                texts = [t[1] for t in text_time_deque]
                first_text = texts[0]
                last_text = texts[-1]


            # Check if at least 3 texts have arrived within the last full second
            if len(text_time_deque) >= 3:
                texts = [t[1] for t in text_time_deque]
                first_text = texts[0]
                last_text = texts[-1]

                # Compute the similarity ratio between the first and last texts
                similarity = SequenceMatcher(None, first_text, last_text).ratio()
                #print(f"Similarity: {similarity:.2f}")

                if similarity > hard_break_even_on_background_noise_min_similarity and len(first_text) > hard_break_even_on_background_noise_min_chars:
                    abrupt_stop = True
                    recorder.stop()

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
                panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
                live.update(panel)
                rich_text_stored = rich_text

            # Mark the task as done
            text_queue.task_done()

    def process_text(text):
        global recorder, full_sentences, prev_text, abrupt_stop
        if IS_DEBUG: print(f"SENTENCE: post_speech_silence_duration: {recorder.post_speech_silence_duration}")
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text)
        text = text.rstrip()
        text_time_deque.clear()
        if text.endswith("..."):
            text = text[:-2]
                
        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        if abrupt_stop:
            abrupt_stop = False
            if relisten_on_abrupt_stop:
                recorder.listen()
                recorder.start()
                if hasattr(recorder, "last_words_buffer"):
                    recorder.frames.extend(list(recorder.last_words_buffer))

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'medium.en',
        #'input_device_index': 1, # mic
        #'input_device_index': 2, # stereomix
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        #'silero_sensitivity': 0.05,
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.05,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': False,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 1,
        'no_log_file': True,
        'initial_prompt': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )
        #'initial_prompt': "Use ellipses for incomplete sentences like: I went to the..."        
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    # Start the worker thread
    worker_thread = threading.Thread(target=process_queue, daemon=True)
    worker_thread.start()

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        # Send sentinel value to worker thread to exit
        text_queue.put(None)
        worker_thread.join()
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)


