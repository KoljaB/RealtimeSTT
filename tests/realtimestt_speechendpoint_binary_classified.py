#IS_DEBUG = True
IS_DEBUG = False
USE_STEREO_MIX = True
LOOPBACK_DEVICE_NAME = "stereomix"
LOOPBACK_DEVICE_HOST_API = 0

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
sentence_end_marks = ['.', '!', '?', '。']


detection_speed = 2.0 # set detection speed between 0.1 and 2.0



if detection_speed < 0.1:
    detection_speed = 0.1
if detection_speed > 2.5:
    detection_speed = 2.5

last_detection_pause = 0
last_prob_complete = 0
last_suggested_pause = 0
last_pause = 0
unknown_sentence_detection_pause = 1.8
ellipsis_pause = 4.5
punctuation_pause = 0.4
exclamation_pause = 0.3
question_pause = 0.2

hard_break_even_on_background_noise = 6
hard_break_even_on_background_noise_min_texts = 3
hard_break_even_on_background_noise_min_chars = 15
hard_break_even_on_background_noise_min_similarity = 0.99

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
    texts_without_punctuation = []
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

    def ends_with_string(text: str, s: str):
        if text.endswith(s):
            return True
        if len(text) > 1 and text[:-1].endswith(s):
            return True
        return False

    def sentence_end(text: str):
        if text and text[-1] in sentence_end_marks:
            return True
        return False

    def additional_pause_based_on_words(text):
        word_count = len(text.split())
        pauses = {
            0: 0.35,
            1: 0.3,
            2: 0.25,
            3: 0.2,
            4: 0.15,
            5: 0.1,
            6: 0.05,
        }
        return pauses.get(word_count, 0.0)
    
    def strip_ending_punctuation(text):
        """Remove trailing periods and ellipses from text."""
        text = text.rstrip()
        for char in sentence_end_marks:
            text = text.rstrip(char)
        return text
    
    def get_suggested_whisper_pause(text):
        if ends_with_string(text, "..."):
            return ellipsis_pause
        elif ends_with_string(text, "."):
            return punctuation_pause
        elif ends_with_string(text, "!"):
            return exclamation_pause
        elif ends_with_string(text, "?"):
            return question_pause
        else:
            return unknown_sentence_detection_pause

    def find_stereo_mix_index():
        import pyaudio
        audio = pyaudio.PyAudio()
        devices_info = ""
        for i in range(audio.get_device_count()):
            dev = audio.get_device_info_by_index(i)
            devices_info += f"{dev['index']}: {dev['name']} (hostApi: {dev['hostApi']})\n"

            if (LOOPBACK_DEVICE_NAME.lower() in dev['name'].lower()
                    and dev['hostApi'] == LOOPBACK_DEVICE_HOST_API):
                return dev['index'], devices_info

        return None, devices_info

    def find_matching_texts(texts_without_punctuation):
        """
        Find entries where text_without_punctuation matches the last entry,
        going backwards until the first non-match is found.
        
        Args:
            texts_without_punctuation: List of tuples (original_text, stripped_text)
            
        Returns:
            List of tuples (original_text, stripped_text) matching the last entry's stripped text,
            stopping at the first non-match
        """
        if not texts_without_punctuation:
            return []
        
        # Get the stripped text from the last entry
        last_stripped_text = texts_without_punctuation[-1][1]
        
        matching_entries = []
        
        # Iterate through the list backwards
        for entry in reversed(texts_without_punctuation):
            original_text, stripped_text = entry
            
            # If we find a non-match, stop
            if stripped_text != last_stripped_text:
                break
                
            # Add the matching entry to our results
            matching_entries.append((original_text, stripped_text))
        
        # Reverse the results to maintain original order
        matching_entries.reverse()
        
        return matching_entries

    def process_queue():
        global recorder, full_sentences, prev_text, displayed_text, rich_text_stored, text_time_deque, abrupt_stop, rapid_sentence_end_detection, last_prob_complete, last_suggested_pause, last_pause
        while True:
            text = None  # Initialize text to ensure it's defined

            try:
                # Attempt to retrieve the first item, blocking with timeout
                text = text_queue.get(timeout=1)
            except queue.Empty:
                continue  # No item retrieved, continue the loop

            if text is None:
                # Exit signal received
                break

            # Drain the queue to get the latest text
            try:
                while True:
                    latest_text = text_queue.get_nowait()
                    if latest_text is None:
                        text = None
                        break
                    text = latest_text
            except queue.Empty:
                pass  # No more items to retrieve

            if text is None:
                # Exit signal received after draining
                break

            text = preprocess_text(text)
            current_time = time.time()
            text_time_deque.append((current_time, text))
            
            # get text without ending punctuation
            text_without_punctuation = strip_ending_punctuation(text)

            # print(f"Text: {text}, Text without punctuation: {text_without_punctuation}")
            texts_without_punctuation.append((text, text_without_punctuation))

            matches = find_matching_texts(texts_without_punctuation)
            #print("Texts matching the last entry's stripped version:")

            added_pauses = 0
            contains_ellipses = False
            for i, match in enumerate(matches):
                same_text, stripped_punctuation = match
                suggested_pause = get_suggested_whisper_pause(same_text)
                added_pauses += suggested_pause
                if ends_with_string(same_text, "..."):
                    contains_ellipses = True
            
            avg_pause = added_pauses / len(matches) if len(matches) > 0 else 0
            suggested_pause = avg_pause
            # if contains_ellipses:
            #     suggested_pause += ellipsis_pause / 2

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

            # pause = new_detection + suggested_pause
            pause = (new_detection + suggested_pause) * detection_speed

            # **Add Additional Pause Based on Word Count**
            # extra_pause = additional_pause_based_on_words(text)
            # pause += extra_pause  # Add the extra pause to the total pause duration

            # Optionally, you can log this information for debugging
            if IS_DEBUG:
                print(f"Prob: {prob_complete:.2f}, "
                    f"whisper {suggested_pause:.2f}, "
                    f"model {new_detection:.2f}, "
                    # f"extra {extra_pause:.2f}, "
                    f"final {pause:.2f} | {transtext} ")

            recorder.post_speech_silence_duration = pause

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

            displayed_text = new_displayed_text
            last_prob_complete = new_detection
            last_suggested_pause = suggested_pause
            last_pause = pause
            panel = Panel(rich_text, title=f"[bold green]Prob complete:[/bold green] [bold yellow]{prob_complete:.2f}[/bold yellow], pause whisper [bold yellow]{suggested_pause:.2f}[/bold yellow], model [bold yellow]{new_detection:.2f}[/bold yellow], last detection [bold yellow]{last_detection_pause:.2f}[/bold yellow]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

            text_queue.task_done()

    def process_text(text):
        global recorder, full_sentences, prev_text, abrupt_stop, last_detection_pause
        last_prob_complete, last_suggested_pause, last_pause
        last_detection_pause = recorder.post_speech_silence_duration
        if IS_DEBUG: print(f"Model pause: {last_prob_complete:.2f}, Whisper pause: {last_suggested_pause:.2f}, final pause: {last_pause:.2f}, last_detection_pause: {last_detection_pause:.2f}")
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
        'model': 'large-v3',
        #'realtime_model_type': 'medium.en',
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.05,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 1,
        'batch_size': 4,
        'realtime_batch_size': 4,
        'no_log_file': True,
        'initial_prompt_realtime': (
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

    if USE_STEREO_MIX:
        device_index, devices_info = find_stereo_mix_index()
        if device_index is None:
            live.stop()
            console.print("[bold red]Stereo Mix device not found. Available audio devices are:\n[/bold red]")
            console.print(devices_info, style="red")
            sys.exit(1)
        else:
            recorder_config['input_device_index'] = device_index
            console.print(f"Using audio device index {device_index} for Stereo Mix.", style="green")

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
