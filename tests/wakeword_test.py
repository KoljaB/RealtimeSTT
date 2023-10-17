from RealtimeSTT import AudioToTextRecorder
import logging

if __name__ == '__main__':

    def recording_started():
        print("Speak now...")

    def recording_finished():
        print("Speech end detected... transcribing...")

    with AudioToTextRecorder(spinner=False, level=logging.DEBUG, model="small.en", language="en", wake_words="jarvis", on_wakeword_detected=recording_started, on_recording_stop=recording_finished
        ) as recorder:
        print('Say "Jarvis" then speak.')
        print(recorder.text())
        print("Done. Now we should exit.")