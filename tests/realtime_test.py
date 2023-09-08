import os
from RealtimeSTT import AudioToTextRecorder

detected_text = ""
displayed_text = ""

def clear_console():
    if os.name == 'posix':  # For UNIX or macOS
        os.system('clear')
    elif os.name == 'nt':  # For Windows
        os.system('cls')

def text_detected(text):
    global displayed_text
    if detected_text + text != displayed_text:
        displayed_text = detected_text + text
        clear_console()
        print(displayed_text)

recorder = AudioToTextRecorder(spinner=False, model="large-v2", language="en", silero_sensitivity=0.2, post_speech_silence_duration=0.4, min_length_of_recording=0.5, min_gap_between_recordings=0.05, realtime_preview_resolution = 0.05, realtime_preview = True, realtime_preview_model = "tiny", on_realtime_preview=text_detected)

print("Say something...")

while (True): 
    detected_text += recorder.text() + " "
    text_detected("")
