#IS_DEBUG = True
IS_DEBUG = False

import os
import re
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
    {'import_name': 'colorama'},
    {'import_name': 'RealtimeSTT'},
    {'import_name': 'transformers'},
    {'import_name': 'torch'},
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
    console = Console()
    console.print("System initializing, please wait")

    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama

    import torch
    import torch.nn.functional as F
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    # Load classification model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "KoljaB/SentenceFinishedClassification"
    max_length = 128

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    classification_model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    classification_model.to(device)
    classification_model.eval()

    # Label mapping
    label_map = {0: "Incomplete", 1: "Complete"}

    # We now want probabilities, not just a label
    def get_completion_probability(sentence, model, tokenizer, device, max_length):
        """
        Return the probability that the sentence is complete.
        """
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        # probabilities is [prob_incomplete, prob_complete]
        # We want the probability of being complete
        prob_complete = probabilities[1]
        return prob_complete

    # We have anchor points for probability to detection mapping
    # (probability, rapid_sentence_end_detection)
    anchor_points = [
        (0.0, 1.0),
        (1.0, 0)
    ]    
    # anchor_points = [
    #     (0.0, 0.4),
    #     (0.5, 0.3),
    #     (0.8, 0.2),
    #     (0.9, 0.1),
    #     (1.0, 0)
    # ]

    def interpolate_detection(prob):
        # Clamp probability between 0.0 and 1.0 just in case
        p = max(0.0, min(prob, 1.0))
        # If exactly at an anchor point
        for ap_p, ap_val in anchor_points:
            if abs(ap_p - p) < 1e-9:
                return ap_val

        # Find where p fits
        for i in range(len(anchor_points) - 1):
            p1, v1 = anchor_points[i]
            p2, v2 = anchor_points[i+1]
            if p1 <= p <= p2:
                # Linear interpolation
                ratio = (p - p1) / (p2 - p1)
                return v1 + ratio * (v2 - v1)

        # Should never reach here if anchor_points cover [0,1]
        return 4.0

    speech_finished_cache = {}

    def is_speech_finished(text):
        # Returns a probability of completeness
        # Use cache if available
        if text in speech_finished_cache:
            return speech_finished_cache[text]
        
        prob_complete = get_completion_probability(text, classification_model, tokenizer, device, max_length)
        speech_finished_cache[text] = prob_complete
        return prob_complete

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    text_queue = queue.Queue()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""
    text_time_deque = deque()

    # Default values
    #rapid_sentence_end_detection = 0.2
    end_of_sentence_detection_pause = 0.3
    unknown_sentence_detection_pause = 0.8
    mid_sentence_detection_pause = 2.0
    hard_break_even_on_background_noise = 3.0
    hard_break_even_on_background_noise_min_texts = 3
    hard_break_even_on_background_noise_min_chars = 15
    hard_break_even_on_background_noise_min_similarity = 0.99
    relisten_on_abrupt_stop = True

    abrupt_stop = False
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
        text_queue.put(text)

    def additional_pause_based_on_words(text):
        word_count = len(text.split())
        pauses = {
            1: 0.6,
            2: 0.5,
            3: 0.4,
            4: 0.3,
            5: 0.2,
            6: 0.1,
        }
        return pauses.get(word_count, 0.0)

    def process_queue():
        global recorder, full_sentences, prev_text, displayed_text, rich_text_stored, text_time_deque, abrupt_stop, rapid_sentence_end_detection

        while True:
            try:
                text = text_queue.get(timeout=1)
            except queue.Empty:
                continue

            if text is None:
                # Exit
                break

            text = preprocess_text(text)
            current_time = time.time()

            sentence_end_marks = ['.', '!', '?', '。']

            if text.endswith("..."):
                suggested_pause = mid_sentence_detection_pause
            elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
                suggested_pause = end_of_sentence_detection_pause
            else:
                suggested_pause = unknown_sentence_detection_pause

            prev_text = text
            import string
            transtext = text.translate(str.maketrans('', '', string.punctuation))

            # **Stripping Trailing Non-Alphabetical Characters**
            # Instead of removing all punctuation, we only strip trailing non-alphabetic chars.
            # Use regex to remove trailing non-alphabetic chars:
            cleaned_for_model = re.sub(r'[^a-zA-Z]+$', '', transtext)

            prob_complete = is_speech_finished(cleaned_for_model)

            # Interpolate rapid_sentence_end_detection based on prob_complete
            new_detection = interpolate_detection(prob_complete)

            pause = new_detection + suggested_pause
            # **Add Additional Pause Based on Word Count**
            extra_pause = additional_pause_based_on_words(text)
            pause += extra_pause  # Add the extra pause to the total pause duration

            # Optionally, you can log this information for debugging
            if IS_DEBUG:
                print(f"Prob: {prob_complete:.2f}, "
                    f"whisper {suggested_pause:.2f}, "
                    f"model {new_detection:.2f}, "
                    f"extra {extra_pause:.2f}, "
                    f"final {pause:.2f} | {transtext} ")

            recorder.post_speech_silence_duration = pause

            #if IS_DEBUG: print(f"Prob complete: {prob_complete:.2f}, pause whisper {suggested_pause:.2f}, model {new_detection:.2f}, final {pause:.2f} | {transtext} ")

            text_time_deque.append((current_time, text))

            # Remove old entries
            while text_time_deque and text_time_deque[0][0] < current_time - hard_break_even_on_background_noise:
                text_time_deque.popleft()

            # Check for abrupt stops (background noise)
            if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
                texts = [t[1] for t in text_time_deque]
                first_text = texts[0]
                last_text = texts[-1]
                similarity = SequenceMatcher(None, first_text, last_text).ratio()

                if similarity > hard_break_even_on_background_noise_min_similarity and len(first_text) > hard_break_even_on_background_noise_min_chars:
                    abrupt_stop = True
                    recorder.stop()

            rich_text = Text()
            for i, sentence in enumerate(full_sentences):
                style = "yellow" if i % 2 == 0 else "cyan"
                rich_text += Text(sentence, style=style) + Text(" ")

            if text:
                rich_text += Text(text, style="bold yellow")

            new_displayed_text = rich_text.plain

            if new_displayed_text != displayed_text:
                displayed_text = new_displayed_text
                panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
                live.update(panel)
                rich_text_stored = rich_text

            text_queue.task_done()

    def process_text(text):
        global recorder, full_sentences, prev_text, abrupt_stop
        #if IS_DEBUG: print(f"SENTENCE: post_speech_silence_duration: {recorder.post_speech_silence_duration}")
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

    recorder_config = {
        'spinner': False,
        'model': 'large-v2',
        'realtime_model_type': 'medium.en',
        'language': 'en',
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
        'beam_size_realtime': 3,
        'no_log_file': True,
        'initial_prompt': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)

    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    worker_thread = threading.Thread(target=process_queue, daemon=True)
    worker_thread.start()

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        text_queue.put(None)
        worker_thread.join()
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)
