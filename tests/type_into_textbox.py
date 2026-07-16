from RealtimeSTT import AudioToTextRecorder
import pyautogui

def process_text(text):
    pyautogui.typewrite(text + " ")

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)