from RealtimeSTT import AudioToTextRecorder

def recording_started():
    print("Speak now...")

def recording_finished():
    print("Speech end detected... transcribing...")

recorder = AudioToTextRecorder(spinner=False, model="small.en", language="en", wake_words="jarvis", on_wakeword_detected=recording_started, on_recording_stop=recording_finished)

print('Say "Jarvis" then speak.')
print(recorder.text())