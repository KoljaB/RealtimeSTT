print("Initializing.")

from RealtimeSTT import AudioToTextRecorder
import os
import colorama
import logging
import traceback
from colorama import Fore, Back, Style
colorama.init()

full_sentences = []
displayed_text = ""

def clear_console():
    logging.debug('Clearing console def clear_console():')
    os.system('clear' if os.name == 'posix' else 'cls')

def text_detected(text):
    global displayed_text
    logging.debug('Processing detected text def text_detected(text)')
    sentences_with_style = [
        f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
        for i, sentence in enumerate(full_sentences)
    ]
    new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

    if new_text != displayed_text:
        displayed_text = new_text
        clear_console()
        print(displayed_text)

recorder_config = {
    'spinner': False,
    'model': 'large-v2',
    'language': 'en',
    'silero_sensitivity': 0.01,
    'webrtc_sensitivity': 3,
    'post_speech_silence_duration': 0.6,
    'min_length_of_recording': 0.2,
    'min_gap_between_recordings': 0,
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0,
    'realtime_model_type': 'small.en',
    'on_realtime_transcription_stabilized': text_detected,
}

recorder = AudioToTextRecorder(**recorder_config)

print("Say something...")

while True:
    logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.debug('Wait for text')
    full_sentences.append(recorder.text())
    text_detected("")
